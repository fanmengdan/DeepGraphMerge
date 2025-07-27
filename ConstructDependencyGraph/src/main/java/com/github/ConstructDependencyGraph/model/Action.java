package com.github.ConstructDependencyGraph.model;

import com.github.ConstructDependencyGraph.model.constant.Operation;

/** One of the semantic change actions of the diff hunk */
public class Action {
  private Operation operation;
  private String typeFrom = "";
  private String labelFrom = "";

  // if modify the type or label
  private String typeTo = "";
  private String labelTo = "";

  public Action(Operation operation, String typeFrom, String labelFrom) {
    this.operation = operation;
    this.typeFrom = typeFrom;
    this.labelFrom = labelFrom.trim();
    this.typeTo = "";
    this.labelTo = "";
  }

  public Action(
      Operation operation, String typeFrom, String labelFrom, String typeTo, String labelTo) {
    this.operation = operation == null ? Operation.UKN : operation;
    this.typeFrom = typeFrom == null ? "" : typeFrom;
    this.labelFrom = labelFrom == null ? "" : labelFrom.trim();
    this.typeTo = typeTo == null ? "" : typeTo;
    this.labelTo = labelTo == null ? "" : labelTo.trim();
  }

  @Override
  public String toString() {
    StringBuilder builder = new StringBuilder();
    builder.append(operation);
    builder
        .append(typeFrom.isEmpty() ? "" : " " + typeFrom)
        .append(labelFrom.isEmpty() ? "" : " \"" + labelFrom + "\"");
    if (!typeFrom.equals(typeTo)) {
      builder.append(typeTo.isEmpty() ? "" : " To " + typeTo);
      if (!labelFrom.equals(labelTo)) {
        builder.append(labelTo.isEmpty() ? "" : ": \"" + labelTo + "\"");
      }
    } else {
      if (!labelFrom.equals(labelTo)) {
        builder.append(labelTo.isEmpty() ? "" : " To: \"" + labelTo + "\"");
      }
    }

    builder.append(".");

    return builder.toString();
  }

  @Override
  public boolean equals(Object obj) {
    if (obj == this) {
      return true;
    }

    if (!(obj instanceof Action)) {
      return false;
    }

    Action a = (Action) obj;
    return a.operation.equals(this.operation)
        && a.typeFrom.equals(this.typeFrom)
        && a.typeTo.equals(this.typeTo)
        && a.labelFrom.equals(this.labelFrom)
        && a.labelTo.equals(this.labelTo);
  }

}
